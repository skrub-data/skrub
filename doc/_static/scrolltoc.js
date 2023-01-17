function updateSideBarPosition(sections) {
    var pos = $(window).scrollTop();

    // Highlight the current section
    i = 0;
    current_section = 0;
    $('a.internal').removeClass('active');
    $('ul.active').removeClass('active');
    $('li.preactive').removeClass('preactive');
    for(i in sections) {
        if(sections[i] > pos) {
            break
        }
	console.log(i);
	current_section = i
        if($('a.internal[href$="' + i + '"]').is(':visible')){
            current_section = i
        }
    }
    $('a.internal[href$="' + current_section + '"]').addClass('active');
    $('a.internal[href$="' + current_section + '"]').parent().parent().addClass('active')
    $('a.internal[href$="' + current_section + '"]').parent().parent().parent().addClass('preactive')
    $('a.internal[href$="' + current_section + '"]').parent().parent().parent().parent().parent().addClass('preactive')
}

$(function() {
    sections = {};
    url = document.URL.replace(/#.*$/, "");

    // Grab positions of our sections
    $('.headerlink').each(function(){
        sections[this.href.replace(url, '')] = $(this).offset().top - 150
    });

    updateSideBarPosition(sections);
    $(window).scroll(function(event) {
        updateSideBarPosition(sections)
    });

    $(window).resize(function(event) {
        updateSideBarPosition(sections)
    });
});
