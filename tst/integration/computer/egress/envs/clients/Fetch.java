public class Fetch {
    public static void main(String[] args) throws Exception {
        var request = java.net.http.HttpRequest.newBuilder(java.net.URI.create(args[0]))
            .header("User-Agent", "agentenv-egress-test").build();
        var response = java.net.http.HttpClient.newHttpClient()
            .send(request, java.net.http.HttpResponse.BodyHandlers.discarding());
        System.out.println(response.statusCode());
    }
}
