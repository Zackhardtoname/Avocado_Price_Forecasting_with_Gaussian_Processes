get_data = function(avocado_type, csv_name, dataset_types=c("raw", "train", "test")) {
  length_dataset_types <- length(dataset_types)
  dataset_names <- vector(mode="list", length=length_dataset_types)
  
  for (i in 1:length_dataset_types) {
    dataset_names[i] <- glue('../../data/{avocado_type}/{dataset_types[i]}/{csv_name}.csv')
  }
  
  datasets = lapply(dataset_names, read.csv)
  lapply(datasets, problems)
  return(datasets)
}