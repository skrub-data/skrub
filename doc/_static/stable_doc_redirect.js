window.onload = function() {
    var stable_page_link = document.getElementById("stable_page_link");
    if (stable_page_link) {
        stable_page_link.href = window.location.href.replace("dev", "stable");
    }
}