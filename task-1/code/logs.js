function saveScrollPosition() {
    sessionStorage.setItem('scrollPos', window.scrollY);
}

function restoreScrollPosition() {
    const scrollPos = sessionStorage.getItem('scrollPos');
    if (scrollPos) {
        window.scrollTo(0, parseInt(scrollPos, 10));
    }
}

function refreshPage() {
    saveScrollPosition();
    location.reload();
}

function displayLastModified() {
    const lastModified = new Date(document.lastModified);
    const formattedDate = lastModified.toLocaleString();
    const lastModifiedElement = document.createElement('div');
    lastModifiedElement.className = 'last-modified';
    lastModifiedElement.textContent = 'Last Modified: ' + formattedDate;
    document.body.insertBefore(lastModifiedElement, document.body.firstChild);
}

window.onload = function() {
    displayLastModified();
    restoreScrollPosition();
    setInterval(refreshPage, 5000);
};

// Save the scroll position before the page is unloaded
window.addEventListener('beforeunload', function() {
    if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight) {
        localStorage.setItem('scrollPosition', 'bottom');
    } else {
        localStorage.setItem('scrollPosition', window.scrollY);
    }
});

// Restore the scroll position when the page is loaded
window.addEventListener('load', function() {
    const savedScrollPosition = localStorage.getItem('scrollPosition');
    if (savedScrollPosition === 'bottom') {
        window.scrollTo(0, document.body.scrollHeight);
    } else if (savedScrollPosition !== null) {
        window.scrollTo(0, parseInt(savedScrollPosition, 10));
    }
});


// This script will run when the document is fully loaded
document.addEventListener("DOMContentLoaded", function() {
    // Create an iframe element
    var iframe = document.createElement("iframe");
    var iframe_heading = document.createElement("h1");
    // Set the source of the iframe to error.html
    iframe.src = "profiler_graph.html";
    
    // Set the width and height of the iframe (adjust as needed)
    iframe.style.width = "100%";
    iframe.style.height = "100%";
    iframe.style.maxHeight = "700px"; // You can adjust the height as needed
    iframe.style.border = "none";  // Remove the border if you don't want one
    
    iframe_heading.innerHTML = "⏱️ Time Profile";
    
    // Append the iframe to the body
    document.body.appendChild(iframe_heading);
    document.body.appendChild(iframe);
    
    // Create and append the link at the bottom right corner
    const link = document.createElement('a');
    link.href = 'https://ponderbot.org';
    link.innerText = 'Made with ❤️ at Ponderbot.ORG';
    link.style.position = 'fixed';
    link.style.bottom = '10px';
    link.style.right = '10px';
    link.style.zIndex = '1000';
    link.style.backgroundColor = 'white';
    link.style.padding = '5px';
    link.style.border = '1px solid #ccc';
    link.style.borderRadius = '5px';
    document.body.appendChild(link);
});
