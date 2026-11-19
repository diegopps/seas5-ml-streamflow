# =============================================================================
# load_env.sh
#
# Sourced (not executed) by every job and helper script before it runs
# Python. Runs the commands in the [slurm] env_setup block of config.toml in
# the calling shell, e.g. `module load ...` and `source activate ...`. Does
# nothing if the block is empty.
#
# The block is extracted with awk rather than a TOML parser, because no
# suitable Python may be available until these very commands have run.
# =============================================================================

_s5ml_cfg="$(dirname "${BASH_SOURCE[0]}")/../config.toml"   # pipeline/ -> repo root
_s5ml_setup="$(awk '
    /^env_setup[[:space:]]*=[[:space:]]*"""/ {
        in_block = 1
        sub(/^env_setup[[:space:]]*=[[:space:]]*"""/, "")
        if ($0 ~ /"""/) { sub(/""".*/, ""); print; exit }   # one-line block
        if ($0 !~ /^[[:space:]]*$/) print
        next
    }
    in_block && /"""/ { sub(/""".*/, ""); if ($0 !~ /^[[:space:]]*$/) print; exit }
    in_block { print }
' "${_s5ml_cfg}")"

if [[ -n "${_s5ml_setup//[[:space:]]/}" ]]; then
    eval "${_s5ml_setup}"
fi
unset _s5ml_cfg _s5ml_setup
