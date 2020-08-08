mkdir -p ~/.streamlit/
echo "\
[server]\n\
headledd = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml