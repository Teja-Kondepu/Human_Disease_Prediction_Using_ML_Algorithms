function closeTab() {
    if (confirm("Are you sure you want to exit? This will close the tab.")) {
        fetch('/exit', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.close();
            } else {
                alert('Unable to close the tab. Please close it manually.');
            }
        });
    }
}