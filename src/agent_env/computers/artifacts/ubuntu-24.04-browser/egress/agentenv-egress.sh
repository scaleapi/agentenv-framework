# The egress proxy for a login shell, as the session's supervisord gives every session program (supervisord.conf).
export HTTP_PROXY=http://127.0.0.1:3128 HTTPS_PROXY=http://127.0.0.1:3128 ALL_PROXY=http://127.0.0.1:3128
export http_proxy=http://127.0.0.1:3128 https_proxy=http://127.0.0.1:3128 all_proxy=http://127.0.0.1:3128
export NO_PROXY=localhost,127.0.0.1,::1,host.docker.internal no_proxy=localhost,127.0.0.1,::1,host.docker.internal
export SSL_CERT_FILE=/etc/agentenv/ca-bundle.pem REQUESTS_CA_BUNDLE=/etc/agentenv/ca-bundle.pem
export NODE_EXTRA_CA_CERTS=/etc/agentenv/ca.pem NODE_USE_ENV_PROXY=1
export JAVA_TOOL_OPTIONS="-Dhttp.proxyHost=127.0.0.1 -Dhttp.proxyPort=3128 -Dhttps.proxyHost=127.0.0.1 -Dhttps.proxyPort=3128 -Dhttp.nonProxyHosts=localhost|127.0.0.1|::1|host.docker.internal"
