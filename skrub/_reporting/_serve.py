import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from queue import Empty, Queue
from socketserver import TCPServer
from threading import Thread

BROWSER_TIMEOUT_SECONDS = 3


def open_in_browser(content):
    encoded_content = content.encode("UTF-8")
    queue = Queue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            del args

        def do_GET(self):
            if not self.path.endswith("index.html"):
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(len(encoded_content)))
            self.end_headers()
            self.wfile.write(encoded_content)
            queue.put("done")

    server = TCPServer(("", 0), Handler)
    _, port = server.server_address

    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{port}/index.html"
    webbrowser.open(url)
    try:
        queue.get(timeout=BROWSER_TIMEOUT_SECONDS)
    except Empty:
        raise RuntimeError("Failed to open report in a web browser.")
    server.shutdown()
    server_thread.join()
