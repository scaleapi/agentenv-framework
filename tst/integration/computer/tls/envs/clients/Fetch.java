public class Fetch {
    public static void main(String[] args) throws Exception {
        var connection = (javax.net.ssl.HttpsURLConnection) java.net.URI.create(args[0]).toURL().openConnection();
        System.out.println(connection.getResponseCode());
    }
}
