# Code and Data for [Paper Title]

This repository contains the code and data for reproducing the analysis presented in the scientific paper titled "[Paper Title]", published at [Journal Name].

## Repository Structure

The repository is organized as follows:

- **tables**: Contains the result tables generated during the analysis.
- **LICENSE**: The license file specifying the GPLv3 license for the code in this repository.
- **requirements.txt**: Lists the Python dependencies required to run the scripts.

## Python Scripts

The repository includes several Python scripts that need to be executed in a specific order to reproduce the analysis. Here is the recommended order for running the scripts:

1. **controls_vs_patient.py**: This script compares the control group with the patients, analyzing the differences in their depressive symptoms before and after treatment.

1. **ayahuasca_vs_placebo.py**: This script compares the effects of ayahuasca treatment with a placebo, examining the changes in depressive symptoms.

1. **madrs_brute_force_search_all_variables.py**: IMPORTANT: This script performs an extensive search and analysis of multiple variables related to depression using a brute-force approach. Please note that this script may take weeks to complete running, as it is computationally intensive. It is designed to run in parallel and requires a 30-core machine to execute efficiently.

1. **madrs_brute_force_search_ayahuasca_placebo_selected_variables.py**: This script performs a more focused analysis, specifically targeting the variables related to the ayahuasca treatment and placebo group. It builds upon the results obtained from the previous script.

1. **analysis_of_madrs_brute_force_search_results.py**: This script analyzes the results obtained from the brute-force search, generating additional insights and conclusions based on the data.

Please ensure that you have set up a virtual environment before running the scripts. You can use either virtualenv or poetry to create a virtual environment with the necessary dependencies specified in the requirements.txt file.

## R Script
Additionally, an R script named `scriptaurocbarra.R` and 'wilcoxonanal.R' is included in the repository. This script produces a figure that is relevant to the paper. Please execute this script after running all the Python scripts mentioned above.

## Authors and Dates
Inácio Gomes Medeiros, Isaac Campos Braga

Brain Insitute

Federal University of Rio Grande do Norte

## License
The code in this repository is licensed under the GNU General Public License v3.0 (GPLv3). Please refer to the LICENSE file for more details.

---

By following the instructions provided in this README, you will be able to reproduce the analysis conducted in the paper and generate the result tables and figures.
