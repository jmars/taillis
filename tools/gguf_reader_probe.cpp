// gguf_reader_probe — INVOKE aubier's grouped-expert reader (engine/gguf_model.h),
// the gate-3 instrument for taillis/tools/group_experts.py.
//
// A seat that is never called is indistinguishable from an absent seat: this probe
// does not read the header, it LOADS the file through GGUFModel::load() (which
// parses taillis.expert_groups and rejects malformed groupings at load time) and
// calls GGUFModel::logical() for every logical tensor in the file.
//
// stdout: one line per LOGICAL tensor (group members folded into their logical's
// line):  <name> ne=[..] segs=<n> [lo,hi)<type>:<nbytes> ...   — a flat file and
// its G=1 regrouping must print IDENTICAL stdout (diff empty = the reader sees
// the same segments).  stderr: the load summary.
//
// argv[2], if given, is an expected segment count applied to every expert logical
// tensor (names containing ffn_gate_up_exps / ffn_down_exps): CHECK fails =>
// exit 1, so the expectation can actually fail.  Expert tensors are additionally
// CHECKed to cover [0, N_EXPERT) contiguously.
//
// Build (see aubier/scripts/engine_build.sh; the libs carry a stale RUNPATH into
// the dead /var/data/workspace tree, so LD_LIBRARY_PATH is mandatory at runtime):
//   c++ -O2 -std=c++17 -I <aubier>/src/llama.cpp/ggml/include -I <aubier>/engine \
//       tools/gguf_reader_probe.cpp -L <aubier>/builds/hip/bin \
//       -lggml-base -Wl,-rpath,<aubier>/builds/hip/bin -o builds/gguf_reader_probe

#include "gguf_model.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

static bool is_expert_logical(const std::string & logical) {
    return strstr(logical.c_str(), "ffn_gate_up_exps") != nullptr
        || strstr(logical.c_str(), "ffn_down_exps") != nullptr;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [expect_segs_for_expert_tensors]\n", argv[0]);
        return 2;
    }
    const int expect = (argc >= 3) ? std::atoi(argv[2]) : -1;

    GGUFModel * m = GGUFModel::load(argv[1]);
    std::fprintf(stderr, "loaded %s: %lld file tensors, %zu grouped logicals, key '%s'\n",
                 argv[1], (long long) m->n_tensors(), m->expert_groups.size(),
                 TAILLIS_EXPERT_GROUPS_KEY);

    std::set<std::string> printed;
    for (int64_t i = 0; i < m->n_tensors(); ++i) {
        std::string n = m->tensor_name(i);
        std::string logical = n;
        bool member = false;
        if (logical.size() > 7 && logical.compare(logical.size() - 7, 7, ".weight") == 0)
            logical.resize(logical.size() - 7);
        std::string g;
        if (group_member_logical(logical, &g)) { member = true; logical = g; }
        if (!printed.insert(logical).second) continue;   // members fold into one line

        const std::string qname = member ? logical + ".weight" : n;
        LogicalTensor lt = m->logical(qname.c_str());

        int lo = 0;
        for (const Segment & s : lt.segs) {
            CHECK(s.lo == lo, "%s: segment [%d,%d) breaks contiguity at %d",
                  lt.name.c_str(), s.lo, s.hi, lo);
            CHECK(s.hi > s.lo, "%s: empty segment [%d,%d)", lt.name.c_str(), s.lo, s.hi);
            lo = s.hi;
        }
        if (is_expert_logical(logical)) {
            CHECK(lo == (int) N_EXPERT,
                  "%s: segments cover [0,%d), expected [0,%d)", lt.name.c_str(), lo, (int) N_EXPERT);
            if (expect >= 0)
                CHECK((int) lt.segs.size() == expect,
                      "%s: %zu segments, expected %d", lt.name.c_str(), lt.segs.size(), expect);
        }

        std::printf("%s ne=[%lld,%lld,%lld,%lld] segs=%zu", lt.name.c_str(),
                    (long long) lt.ne[0], (long long) lt.ne[1],
                    (long long) lt.ne[2], (long long) lt.ne[3], lt.segs.size());
        for (const Segment & s : lt.segs)
            std::printf(" [%d,%d)%s:%zu", s.lo, s.hi, ggml_type_name(s.type), s.nbytes);
        std::printf("\n");
    }
    delete m;
    return 0;
}
