import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from queue import Empty, Queue
from socketserver import TCPServer
from threading import Thread

BROWSER_TIMEOUT_SECONDS = 5


def open_in_browser(content):
    """Display content (an HTML page as a string) in a web browser.

    This function starts a local server in a separate thread, then asks the
    webbrowser to open the corresponding URL. Once the server has received and
    answered the browser's request, it shuts down. (Note this means that if the
    user refreshes the page they will get a 'server not found' error.) If the
    browser's request does not arrive after 5 seconds, the server is shut down
    and a RuntimeError is raised saying that opening the content failed.

    An alternative could be to write content in a temporary file and open it in
    the browser. This approach would have several drawbacks:

    - In some systems, firefox cannot open files in /tmp
    - We would have the risk of cluttering the file system with left-behind
      temporary files if the program gets killed before it removes it, or fails
      to do so for some other reason.
    - We would have the risk of removing the file before it gets opened.
    - We would have no way to know if the operation succeeded and the file was
      actually opened.
    """
    encoded_content = content.encode("UTF-8")
    queue = Queue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            del args

        def do_GET(self):
            if not self.path.endswith("index.html"):
                # We point the browser to index.html but it might issue other
                # requests as well, typically for favicon.ico. We ignore those.
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(len(encoded_content)))
            self.end_headers()
            self.wfile.write(encoded_content)
            # The browser has received the content, we can tell the main thread
            # to shut down the server.
            queue.put("done")

    # We let the OS choose the port to make sure the chosen one is free.
    server = TCPServer(("", 0), Handler)
    _, port = server.server_address

    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{port}/index.html"
    webbrowser.open(url)
    try:
        # Wait for the server to put ``"done"`` in the queue, signalling it has
        # sent the content to the browser.
        queue.get(timeout=BROWSER_TIMEOUT_SECONDS)
    except Empty:
        # We reached the timeout without receiving anything from the server's
        # thread -- something went wrong.
        raise RuntimeError("Failed to open report in a web browser.")
    server.shutdown()
    server_thread.join()
