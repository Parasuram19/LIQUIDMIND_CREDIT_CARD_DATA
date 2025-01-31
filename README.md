# Customer Segmentation

This project is a Streamlit application that allows users to predict the customer segment based on various financial features.

## Installation

1. Clone the repository:
```
git clone https://github.com/your-username/customer-segmentation.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```
streamlit run steamlit.py
```
2. The application will open in your default web browser.
3. Enter the customer's financial features in the input fields.
4. Click the "Predict" button to see the predicted customer segment.
5. You can also toggle the "Show Customer Segment Distribution" checkbox to display the pie chart and bar chart of the customer segment distribution.

## API

The application uses the following APIs:

- Streamlit: For building the web application
- Numpy: For numerical operations
- Scikit-learn: For the Random Forest Classifier and StandardScaler
- Pandas: For data manipulation
- Plotly Express: For creating the data visualizations

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them
4. Push your changes to your forked repository
5. Submit a pull request to the main repository

## License

This project is licensed under the [MIT License](LICENSE).

## Testing

To run the tests for this project, use the following command:
```
pytest tests/
```

This will run the test suite and display the results.