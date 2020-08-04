


// Function to handle the search criteria via button click
function handleClickSearch() {

    const query = d3.select('#search_query').property("value");
    console.log(query);

    if (query) {
        console.log(query);
        document.getElementById("search_query").placeholder=query;

        var url = "http://0.0.0.0/api/search/"; // Url of to locate the api on any system (e.g: host_ip:port/route)
        var updated_url = url + query ;

        fetch(updated_url)
          .then(function (response) {
            return response.json();
          })
          .then(function (data) {
                console.log(data);
                articleData = data;
                // buildTable(articleData);
                var num_query_results = articleData.length;

                document.getElementById("search_num").innerHTML = "Your query returned " + num_query_results + " results.";
                console.log(num_query_results);

                var results_container = document.getElementById("result_list") ;

                for (i = 0; i < articleData.length ; i++) {
                  article_record = articleData[i];
                  article_title = article_record['title']; // title
                  article_authors = article_record['authors'];
                  article_urls = article_record['url'];
                  article_url = article_urls.split("; ")[0];
                  results_container.innerHTML = "<div class = \"list_element\"> <p> Title: " + article_title + "</p> \
                                                 <p> Authors: " + article_authors + " </p> <br> \
                                                 <p> Link:  <a href = "+ article_url + "  \" target = blank\"> article link </a> </p> </div>";
                }
                // 



          })

          .catch(function (err) {
            console.log(err);
          });
    }
}


// Attach an event to listen for the search button
d3.select("#search_button").on("click", handleClickSearch);

