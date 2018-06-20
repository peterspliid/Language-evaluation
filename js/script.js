var path = window.location.pathname.split("/");
var session = path.filter(Boolean)[0];

$(document).ready(function() {
    $(".datatable").DataTable();
});