const response = await fetch(process.argv[2], { headers: { "user-agent": "agentenv-egress-test" } });
console.log(response.status);
