


// Function to handle the search criteria via button click
function handleClickSearch() {

    const query = d3.select('#search_query').property("value");

    if (query) {
        console.log(query);
        document.getElementById("search_query").placeholder=query;

        var url = "/api/search/";
        var updated_url = url + query ;

        fetch(updated_url)
          .then(function (response) {
            return response.json();
          })
          .then(function (data) {
                console.log(data);
                recipeData = data;
                // buildTable(recipeData);
                var num_query_results = recipeData.length;

                document.getElementById("search_num").innerHTML = "Your query returned " + num_query_results + " results.";
          })

          .catch(function (err) {
            console.log(err);
          });
    }
}


// Attach an event to listen for the search button
d3.select("#search-btn").on("click", handleClickSearch);
