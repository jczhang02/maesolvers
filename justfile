# Common commands for this repo. `just` lists them.
#
# a100server layout (see CLAUDE.md, Experiments):
#   code    /jczhang02/maesolvers           clone of the bare repo below
#   bare    /jczhang02/git/maesolvers.git   `git push a100server`
#   data    /jczhang02/maesolvers-data      logs/, data/
#   venv    /root/.venvs/maesolvers         outside the code tree
#   mlflow  the research programme's shared server, port 30000
#           (started from monge-ampere-equations, `just mlflow-server` there)

server := "a100server"
server_ip := "10.30.121.100"
repo := "/jczhang02/maesolvers"
bare := "/jczhang02/git/maesolvers.git"
data := "/jczhang02/maesolvers-data"
venv := "/root/.venvs/maesolvers"
port := "30000"
env := "UV_PROJECT_ENVIRONMENT=" + venv + " UV_CACHE_DIR=/jczhang02/.cache/uv MLFLOW_TRACKING_URI=http://127.0.0.1:" + port

default:
    @just --list

# Format and fix Python
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Check formatting, lint and tests, as CI does
check:
    uv run ruff format --check .
    uv run ruff check --no-fix .
    uv run pytest -q

# One-time a100server setup: bare repo, clone, data dirs, local remote
server-init:
    ssh {{ server }} 'set -e; test -d {{ bare }} || git init -q --bare -b main {{ bare }}; mkdir -p {{ data }}/logs {{ data }}/data'
    git remote get-url {{ server }} >/dev/null 2>&1 || git remote add {{ server }} ssh://{{ server }}{{ bare }}
    git push -q {{ server }} main
    ssh {{ server }} 'test -d {{ repo }} || git clone -q {{ bare }} {{ repo }}'

# Push BRANCH (default: current) to a100server and check it out there
push branch=`git branch --show-current`:
    git push -q {{ server }} {{ branch }}
    ssh {{ server }} 'cd {{ repo }} && git fetch -q origin && git checkout -q -B {{ branch }} origin/{{ branch }} && git log --oneline -1'

# a100server reaches PyPI only through the Shanhe mirror, so server-sync
# installs the versions pinned in uv.lock from there instead of `uv sync`.

# Install the locked environment (with the neural extra and mlflow) on a100server
server-sync:
    ssh {{ server }} 'set -e; cd {{ repo }}; export {{ env }}; \
        test -x {{ venv }}/bin/python || uv venv -q --python 3.12 {{ venv }}; \
        uv export -q --frozen --no-hashes --no-emit-project --extra neural --group experiments -o /tmp/maesolvers-requirements.txt; \
        uv pip install -q --python {{ venv }}/bin/python -r /tmp/maesolvers-requirements.txt; \
        uv pip install -q --python {{ venv }}/bin/python --no-deps -e .; \
        {{ venv }}/bin/python -c "import maesolvers, torch, mlflow; print(\"maesolvers\", maesolvers.__file__, \"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"mlflow\", mlflow.__version__)"'

# Run CMD in the a100server checkout, in the project environment (foreground)
remote +cmd:
    ssh {{ server }} 'cd {{ repo }} && {{ env }} uv run --no-sync {{ cmd }}'

# CMD runs inside a double-quoted tmux command: avoid " and $ in it.

# Run CMD on a100server in tmux session maesolvers-NAME; log to data/logs/NAME.log
remote-bg name +cmd:
    ssh {{ server }} 'cd {{ repo }} && tmux new-session -d -s maesolvers-{{ name }} "{{ env }} uv run --no-sync {{ cmd }} 2>&1 | tee -a {{ data }}/logs/{{ name }}.log"'
    @echo "attach: ssh -t {{ server }} tmux attach -t maesolvers-{{ name }}; log: {{ data }}/logs/{{ name }}.log"

# URL of the shared MLflow UI
mlflow-ui:
    @echo "http://{{ server_ip }}:{{ port }}"
