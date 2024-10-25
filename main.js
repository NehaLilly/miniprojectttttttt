const fileInput = document.querySelector(".file-input"),
    progressArea = document.querySelector(".progress-area"),
    uploadedArea = document.querySelector(".uploaded-area"),
    form = document.querySelector("form");

// When user clicks anywhere on the form, trigger the file input click
form.addEventListener("click", () => {
    fileInput.click();
});

fileInput.onchange = ({ target }) => {
    let file = target.files[0]; // Get the selected file
    if (file) {
        let fileName = file.name;
        if (fileName.length >= 12) {
            let splitName = fileName.split('.');
            fileName = splitName[0].substring(0, 13) + "... ." + splitName[1]; // Shorten the name if it's too long
        }
        uploadFile(file, fileName); // Upload the file and show progress
    }
}

function uploadFile(file, name) {
    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload"); // Replace with your actual upload endpoint in Node.js

    xhr.upload.addEventListener("progress", ({ loaded, total }) => {
        let fileLoaded = Math.floor((loaded / total) * 100); // Calculate percentage
        let fileTotal = Math.floor(total / 1000); // Convert size to KB
        let fileSize;
        (fileTotal < 1024) ? fileSize = fileTotal + " KB" : fileSize = (fileTotal / 1024).toFixed(2) + " MB"; // Convert size to MB if over 1MB

        // HTML for showing the progress
        let progressHTML = `<li class="row">
                                <i class="fas fa-file-alt"></i>
                                <div class="content">
                                    <div class="details">
                                        <span class="name">${name} • Uploading</span>
                                        <span class="percent">${fileLoaded}%</span>
                                    </div>
                                    <div class="progress-bar">
                                        <div class="progress" style="width: ${fileLoaded}%"></div>
                                    </div>
                                </div>
                            </li>`;
        
        progressArea.innerHTML = progressHTML; // Show progress

        // When the upload is complete
        if (loaded === total) {
            progressArea.innerHTML = ""; // Clear progress area

            // HTML for showing the uploaded file
            let uploadedHTML = `<li class="row">
                                    <div class="content upload">
                                        <i class="fas fa-file-alt"></i>
                                        <div class="details">
                                            <span class="name">${name} • Uploaded</span>
                                            <span class="size">${fileSize}</span>
                                        </div>
                                    </div>
                                    <i class="fas fa-check"></i>
                                </li>`;
            uploadedArea.insertAdjacentHTML("afterbegin", uploadedHTML); // Show uploaded file
        }
    });

    xhr.send(formData); // Send the file via the formData
}
