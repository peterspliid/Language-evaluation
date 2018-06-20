$(document).ready(function(){
    get_status();
});

function get_status() {
    $.get('check', function(response) {
        if (response == 'ok') {
            location.reload();
        } else {
            setTimeout(get_status, 5000);
        }
    });
}