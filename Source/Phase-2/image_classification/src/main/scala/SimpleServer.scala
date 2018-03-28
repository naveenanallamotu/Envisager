
import java.io.{File, ByteArrayInputStream}
import java.nio.file.{Files, Paths}
import javax.imageio.{ImageWriteParam, IIOImage, ImageIO}
import java.net.InetSocketAddress


import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}
import sun.misc.BASE64Decoder




object SimpleHttpServer extends App{
  val server = HttpServer.create(new InetSocketAddress(8080), 0)
  server.createContext("/get_custom", new RootHandler())
  server.setExecutor(null)
  server.start()
  println("------ waiting for Request ------")
}

class RootHandler extends HttpHandler {
  def handle(httpExchange: HttpExchange) {
    val data = httpExchange.getRequestBody
    val imageByte = (new BASE64Decoder()).decodeBuffer(data);
    val bytes = new ByteArrayInputStream(imageByte)
    val image = ImageIO.read(bytes)
    ImageIO.write(image, "png", new File("image.png"))
    println("------ Image receiving complete ------")


    val res = classification.classifyImage("Image.png");

    httpExchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*")
    httpExchange.sendResponseHeaders(200, res.length())
    val outStream = httpExchange.getResponseBody
    outStream.write(res.getBytes)
    outStream.close()
  }
}
