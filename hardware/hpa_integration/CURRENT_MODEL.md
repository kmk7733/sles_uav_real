# Current approved HPA model

Candidate: v4_lr1e3_wd1_b64_es10_s1_e40; best epoch35.
Checkpoint: d05e6dc4b0cb410eea1cf2f274946481244ad1fa6c23cc77856d1201c9d2edc2.

Default bundle: /home/rogx/catkin_ws/src/deploy/rogx_hpa_v4
Current shadow/replay release: /home/rogx/hpa_current
Versioned source: /home/rogx/hpa_v4_refresh_5dadd0b_20260923

Read /home/rogx/hpa_current/README.md for validation and recovery.
Old dated directories are historical records. Existing fly.sh/data collection
continue their existing HAA path. This model replacement does not enable flight.

Backup: /home/rogx/backups/hpa-refresh-5dadd0b-20260923/src-before.tar.gz
Rollback: python3 /home/rogx/backups/hpa-refresh-5dadd0b-20260923/activate_model.py --rollback
