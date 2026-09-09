# Invalidated recovery collection

This run is excluded from every recovery analysis.

- Protocol gate violated: shared-cgroup `oom_kill` increased from 18 to 20.
- The increase coincided with five unrelated `run_open_world_audit` workers
  entering the same 8 GiB cgroup and terminating the policy server/client.
- The manifest remained `status=collecting`; 58 arm files formed 29 complete
  states, and no arm file was written for the next state.
- No outcome from this directory may be combined with the clean formal shards.
- No intervention, sample, threshold, noise rule, endpoint, or statistical test
  was changed after this invalidation.
