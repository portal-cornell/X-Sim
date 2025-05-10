window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
      initialSlide: 1,
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
      pagination: false,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // $('.slider-item').width("550px").css("display","flex").css("align-items","center")
    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
        var index = (state['next']+1)
        if(index == 0) {
          index = 6
        }
        if(index == 7) {
          index = 1
        }
        // Set the width of the slider items so that they cover the width of the carousel

        // $('.slider-item').width("300px").css("display","flex").css("align-items","center")
        // $('.slider-item[data-slider-index='+index+']').width("525px").css("display","flex").css("align-items","center")

        // store width of carousel in next lines
        var width = $('.carousel').width();
        console.log(width);

        var center = width / 2;
        var side = (width - center) / 2;

        console.log(center);
        console.log(side);

        $('.slider-item').width(`${side}px`).css("display","flex").css("align-items","center")
        // $('.slider-item').width("384px").css("display","flex").css("align-items","center")
        $('.slider-item[data-slider-index='+index+']').width(`${center}px`).css("display","flex").css("align-items","center")
        // $('.slider-item').width("384px").css("display","flex").css("align-items","center")
        // $('.slider-item[data-slider-index='+index+']').width("550px").css("display","flex").css("align-items","center")
    	});
      carousels[i].on('after:show', state => {
        console.log(state);
        var index = (state['index'])
        if(index == 0) {
          index = 6
        }
        // $('.slider-item').width("300px").css("display","flex").css("align-items","center")
        // $('.slider-item[data-slider-index='+index+']').width("550px").css("display","flex").css("align-items","center")

        // $('.slider-item').width("384px").css("display","flex").css("align-items","center")
        // $('.slider-item[data-slider-index='+index+']').width("550px").css("display","flex").css("align-items","center")

      });


    }


    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    // bulmaSlider.attach();
    $(".slider-navigation-next").click()
    // $(".slider-navigation-next").click()
    // bulmaSlider.attach();
    // $(".slider-navigation-next").click()
})
