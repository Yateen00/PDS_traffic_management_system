## File Structure and Workflow

### Output Files

- **CSV Files**: During training, CSV files containing data for each episode are saved in the `outputs/2way-single-intersection` directory. Currently, there is no command option to change the output location of these CSV files.
- **Plot Files**: For result visualization, the `resultplot` script reads the CSV files from `outputs/2way-single-intersection` and generates plots in the `outputs/plots` directory. The location for plot outputs can be modified using specific command options.

### Testing and Training Workflow

1. **Training**: Run the training process, which will output CSV data in `outputs/2way-single-intersection`.
2. **Plotting**: Use the `resultplot` script to visualize the training data. After plotting, consider moving both the CSV files and generated plots to a `train-plot` directory for better organization.
3. **Testing**: Similarly, after testing, move the CSV and plot files to a `test-plot` directory to clearly separate training and testing outputs.

> **Note**: For flexible file organization, remember to move files as needed between directories like `train-plot` and `test-plot`. Only plot output locations can be changed through commands; the CSV file location is fixed.
