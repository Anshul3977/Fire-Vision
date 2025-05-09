$(document).ready(function() {
    $('#uploadBtn').click(function() {
        var formData = new FormData($('#uploadForm')[0]);
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                console.log('Upload successful:', response);
                $('#logs').text('Video processing started...');
            },
            error: function(error) {
                console.log('Upload error:', error);
                $('#logs').text('Error uploading video.');
            }
        });
    });

    $('#stopBtn').click(function() {
        socket.emit('stop_processing');
        console.log('Stop processing triggered');
        $('#logs').text('Processing stopped.');
    });
});