#!/bin/bash
# The marker scrub, as the session user: a marked bash's own time (printf %()T is bash's builtin, in the process
# itself), its direct child's and its grandchild's, against the universe's (an unmarked date) and real (/proc/uptime-free:
# printed by the root driver around it). Prints "label epoch_seconds".
echo "unmarked_date $(date +%s.%N)"
env AGENTENV_REAL_TIME=1 bash -c 'printf "marked_bash_itself %(%s)T\n" -1; echo "marked_bash_child_date $(date +%s.%N)"; echo "marked_bash_grandchild_sh_date $(sh -c "date +%s.%N")"; echo "marked_bash_exec_child $(bash -c "exec date +%s.%N")"; echo "marked_bash_environ $(tr "\0" "\n" < /proc/$$/environ | grep -c "^AGENTENV_REAL_TIME=1$")"; echo "child_environ $(sh -c "tr \"\\0\" \"\\n\" < /proc/self/environ | grep -c ^AGENTENV_REAL_TIME")"'
env AGENTENV_REAL_TIME=1 python3 -c 'import subprocess,time; print("marked_python_itself", time.time()); print("marked_python_child", subprocess.run(["date","+%s.%N"],capture_output=True,text=True).stdout.strip())'
