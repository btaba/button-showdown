
tmux new -s session_name

cd ~/button-showdown
uwsgi --socket 0.0.0.0:8080 --protocol=http --callable app -w wsgi 
