$(document).ready(function () {
    const socket = io({
        transports: ["websocket", "polling"],
        upgrade: true,
    });
    window.socket = socket;

    function setLogs(text) {
        $("#logs").text(text);
    }

    function addLog(text) {
        var current = $("#logs").text();
        $("#logs").text(current ? current + "\n" + text : text);
    }

    $.getJSON("/health", function (data) {
        if (data.alerts === "twilio") {
            $("#alertMode").text("Twilio is configured: one SMS and one call per fire.");
        } else {
            $("#alertMode").text("Demo mode: alerts are logged on screen and are not sent.");
        }
        $("#limits").text(
            " Uploads up to " + data.max_upload_mb + " MB and " +
            data.max_duration_sec + "s. Frames are skipped (stride " + data.frame_stride +
            ") and the long edge is capped at " + data.max_edge + "px."
        );
    });

    $("#uploadBtn").click(function () {
        var formData = new FormData($("#uploadForm")[0]);
        setLogs("Uploading…");
        $.ajax({
            url: "/upload",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                setLogs(response.message || "Processing started.");
            },
            error: function (xhr) {
                var msg = "Error uploading video.";
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    msg = xhr.responseJSON.error;
                }
                setLogs(msg);
            },
        });
    });

    $("#sampleBtn").click(function () {
        setLogs("Starting the bundled sample…");
        $.ajax({
            url: "/demo",
            type: "POST",
            success: function (response) {
                setLogs(response.message || "Sample started.");
            },
            error: function (xhr) {
                var msg = "Could not start the sample.";
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    msg = xhr.responseJSON.error;
                }
                setLogs(msg);
            },
        });
    });

    $("#stopBtn").click(function () {
        socket.emit("stop_processing");
        setLogs("Stopping…");
    });

    socket.on("status", function (msg) {
        addLog(msg);
    });

    socket.on("stable_update", function (data) {
        const stableFrames = document.getElementById("stableFrames");
        if (stableFrames) {
            stableFrames.src = "data:image/jpeg;base64," + data;
        }
    });

    socket.on("yolo_update", function (data) {
        const frame = document.getElementById("fireAnalysis");
        if (frame) {
            frame.src = "data:image/jpeg;base64," + data;
        }
    });

    socket.on("analysis_update", function (data) {
        const graphElement = document.getElementById("analysis_graph");
        const totalAreaElement = document.getElementById("total_area");
        const areaIncreaseElement = document.getElementById("area_increase");
        if (graphElement && totalAreaElement && areaIncreaseElement && data.graph) {
            graphElement.src = "data:image/png;base64," + data.graph;
            totalAreaElement.innerText = Number(data.total_area).toFixed(2);
            areaIncreaseElement.innerText = Number(data.area_increase).toFixed(2);
        }
    });

    socket.on("analysis_final", function (data) {
        const graphElement = document.getElementById("analysis_graph");
        const totalAreaElement = document.getElementById("total_area");
        if (graphElement && totalAreaElement && data.graph) {
            graphElement.src = "data:image/png;base64," + data.graph;
            totalAreaElement.innerText = Number(data.total_area).toFixed(2);
        }
    });
});
