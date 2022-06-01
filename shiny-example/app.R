library(shiny)
library(dplyr)
library(ggplot2)

potasyum_data <-
  data.frame(
    date = seq(as.Date("2019/6/1"), by = "day", length.out = 30),
    value_name = rep("Potasyum", 30),
    result = sample(1:100, 30)
  )

protein_data <-
  data.frame(
    date = seq(as.Date("2019/6/1"), by = "day", length.out = 30),
    value_name = rep("Protein", 30),
    result = sample(1:100, 30)
  )

stack_data <- rbind(potasyum_data, protein_data)

ui <-
  fluidPage(titlePanel(h2("Blood Test Result System", align = "center")),
            sidebarLayout(
              sidebarPanel(
                selectInput(
                  inputId = "dataset",
                  label = "Choose a blood value:",
                  choices = c("Potasyum", "Protein"),
                  selected = "Protein"
                )
              ),
              mainPanel(plotOutput("ts_plot"),
                        verbatimTextOutput("summary"))
            ))

server <- shinyServer(function(input, output) {
  datasetInput <- reactive({
    stack_data %>% filter(value_name == input$dataset)
  })
  
  
  # Generate a summary of the dataset ----
  output$summary <- renderPrint({
    dataset <- datasetInput()
    summary(dataset$result)
  })
  
  # plot time series
  output$ts_plot <- renderPlot({
    dataset <- datasetInput()
    ggplot(dataset, aes(x = date, y = result)) + geom_line()
    
  })
})



shiny::shinyApp(ui = ui, server = server)