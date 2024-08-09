// menu.js
document.addEventListener('DOMContentLoaded', function() {
    var menuButton = document.querySelector('.menu-button');
    var menu = document.querySelector('.menu');
    var closeButton = document.querySelector('.close-button');

    menuButton.addEventListener('click', function() {
        menu.classList.toggle('active');
    });

    closeButton.addEventListener('click', function() {
        menu.classList.remove('active');
    });
});
