document.addEventListener("DOMContentLoaded", function () {
    const movies = JSON.parse(document.getElementById('movie-data').textContent);
    let currentIndex = 0;
    const prevImg = document.querySelector("#prev-img");
    const currentImg = document.querySelector("#current-img");
    const nextImg = document.querySelector("#next-img");
    const prevTitle = document.querySelector("#prev-title");
    const currentTitle = document.querySelector("#current-title");
    const nextTitle = document.querySelector("#next-title");
    const prevBtn = document.querySelector("#prev-btn");
    const nextBtn = document.querySelector("#next-btn");
    const placeholderImgUrl = document.getElementById('placeholder-url').textContent;

    function setImageSrc(img, movie, title) {
        const imgUrl = `https://image.tmdb.org/t/p/w500${movie.poster_path}`;
        console.log("Image URL:", imgUrl); // Add this line for debugging
        img.onerror = function() {
            img.src = placeholderImgUrl;
        };
        img.src = imgUrl;
        img.alt = movie.title;
        title.textContent = movie.title;
    }
    

    function updateImages() {
        setImageSrc(prevImg, movies[(currentIndex - 1 + movies.length) % movies.length], prevTitle);
        setImageSrc(currentImg, movies[currentIndex], currentTitle);
        setImageSrc(nextImg, movies[(currentIndex + 1) % movies.length], nextTitle);
    }

    prevBtn.onclick = () => {
        currentIndex = (currentIndex - 1 + movies.length) % movies.length;
        updateImages();
    };

    nextBtn.onclick = () => {
        currentIndex = (currentIndex + 1) % movies.length;
        updateImages();
    };

    updateImages();
});
