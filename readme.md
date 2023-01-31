# Implicit Temporal Reasoning for Evidence-Based Fact-Checking

This is the code for the 'Implicit Temporal Reasoning for Evidence-Based Fact-Checking' paper.

## Dependencies

|Package|Version|
|------------------------|---------|
|allennlp|2.10.0|
|attrs|22.1.0|
|base58|2.1.1|
|blis|0.7.8|
|boto3|1.24.90|
|botocore|1.27.90|
|cached-path|1.1.6|
|cachetools|5.2.0|
|catalogue|2.0.8|
|certifi|2022.9.24|
|charset-normalizer|2.1.1|
|click|8.1.3|
|colorama|0.4.5|
|commonmark|0.9.1|
|contourpy|1.0.5|
|cycler|0.11.0|
|cymem|2.0.6|
|dill|0.3.5.1|
|docker-pycreds|0.4.0|
|fairscale|0.4.6|
|filelock|3.7.1|
|fonttools|4.37.4|
|gitdb|4.0.9|
|GitPython|3.1.29|
|google-api-core|2.8.2|
|google-auth|2.12.0|
|google-cloud-core|2.3.2|
|google-cloud-storage|2.5.0|
|google-crc32c|1.5.0|
|google-resumable-media|2.4.0|
|googleapis-common-protos|1.56.4|
|h5py|3.7.0|
|huggingface-hub|0.10.1|
|idna|3.4|
|iniconfig|1.1.1|
|Jinja2|3.1.2|
|jmespath|1.0.1|
|joblib|1.2.0|
|kiwisolver|1.4.4|
|langcodes|3.3.0|
|lmdb|1.3.0|
|lxml|4.9.1|
|MarkupSafe|2.1.1|
|matplotlib|3.6.1|
|more-itertools|8.14.0|
|murmurhash|1.0.8|
|nltk|3.7|
|numpy|1.23.1|
|packaging|21.3|
|pandas|1.5.0|
|pathtools|0.1.2|
|pathy|0.6.2|
|Pillow|9.2.0|
|pip|21.3.1|
|pluggy|1.0.0|
|preshed|3.0.7|
|promise|2.3|
|protobuf|3.20.0|
|psutil|5.9.2|
|py|1.11.0|
|pyasn1|0.4.8|
|pyasn1-modules|0.2.8|
|pydantic|1.8.2|
|Pygments|2.13.0|
|pyparsing|3.0.9|
|pytest|7.1.3|
|python-dateutil|2.8.2|
|pytz|2022.4|
|PyYAML|6.0|
|regex|2022.9.13|
|requests|2.28.1|
|rich|12.1.0|
|rsa|4.9|
|s3transfer|0.6.0|
|sacremoses|0.0.53|
|scikit-learn|1.1.2|
|scipy|1.9.0|
|sentencepiece|0.1.97|
|sentry-sdk|1.9.10|
|setproctitle|1.3.2|
|setuptools|60.2.0|
|shortuuid|1.0.9|
|six|1.16.0|
|smart-open|5.2.1|
|smmap|5.0.0|
|spacy|3.3.1|
|spacy-legacy|3.0.10|
|spacy-loggers|1.0.3|
|srsly|2.4.4|
|tensorboardX|2.5.1|
|termcolor|1.1.0|
|thinc|8.0.17|
|threadpoolctl|3.1.0|
|tokenizers|0.12.1|
|tomli|2.0.1|
|torch|1.11.0|
|torchvision|0.12.0|
|tqdm|4.64.1|
|traitlets|5.4.0|
|transformers|4.20.1|
|typer|0.4.2|
|typing_extensions|4.4.0|
|urllib3|1.26.12|
|wandb|0.12.21|
|wasabi|0.10.1|
|wheel|0.37.1|

## Content

- Folder `division1DifferencePublication/` contains all the code regarding the division
of the claims/evidence into time buckets according to their publication time.

- Folder `division2DifferenceTimeText/` contains all the code for the division of the
claim/evidence into time buckets according to the publication time of the claim
and the time references in the text of the claim/evidence.

- Folder `division1And2/` contains all the code about using both the division of the
claim/evidence according to their publication time (division 1) and the time-
in text reference (division 2).

- Files `Experiment1And2.py` and `Experiment1And2BERT.py` calculate the Spearman's correlation
between the rankings of the evidence relevance o_i and preferred verification
label q_i for respectively all BiLSTM and DistilRoBERTA models (Impact on
Evidence and Label Scores).

- Files `spearmanRankTime.py` and `spearmanRankTimeBERT.py` calculate intra and inter SpearmanRankingCoefficients
for division1DifferencePublication respectively for a BiLSTM and DistilROBERTA model as encoder.

- Files `spearmanRankAbsolute.py` and `spearmanRankAbsoluteBERT.py` calculate intra and inter SpearmanRankingCoefficients
for division1DifferenceTimeText respectively for a BiLSTM and DistilROBERTA model as encoder.

- Files `spearmanRankEverything.py` and `spearmanRankEverythingBERT.py` calculate intra and inter SpearmanRankingCoefficients
for division1And2 respectively for a BiLSTM and DistilROBERTA model as encoder.

- Files `attributionTime.py`, `attributionTimeText.py` and `attributionEverything.py` calculate the attribution for the
claim, evidence and temporal information features for the various BiLSTM models using integrated gradients.

## Data

We use the [Multi-FC](https://aclanthology.org/D19-1475/) dataset for training and evaluating the models. This data could be downloaded from https://competitions.codalab.org/competitions/21163 
Do the following things:
- Copy the `snippets` folder to the root of the project. 
- Copy `dev.tsv` and `train.tsv` to the already existed `dev` and `train` folder.
- The `test.tsv` file doesn´t contain verification labels. For this a `test.tsv` file is already available in the `test` folder where the labels are extracted from the crafting project of the `Multi-FC` dataset ([Multi-FC Scrapper](https://github.com/lcschv/MultiFCscrapper)). Another option is to participate in the CodaLab competition.

## Preprocessing of the data
In this section we describe with preprocessing of the data. For running the code in this section and the next section, it is assumed that those are run from **the root of the project**.
### Open Information Extraction
For doing open information extraction. Run the file `OpenInformationExtraction.py` with as first argument the type of dataset: Dev, Train and Test (called mode in the code) and as second argument the path to the dataset. This would create a folder `OpenInformation` if it isn't already made and the open information extraction of the claim and evidence snippets will be stored in that folder. E.g.for extracting the open information of the claim and evidence snippets of the Train dataset, run the following command:

```sh
python OpenInformationExtraction.py Train train.tsv
```
### Editing article text of claim and snippet 
The file `writeTextToDocument.py` contains two editing functions:
- editArticleText: Do segmentation of the text in sentences
- editArticleTextWithUppercaseEditing: Do beside the segmentqtion of the the text in sentences, also editing of the uppercase letters with the use of Open information extraction.

Those two functions have 4 arguments `mode`, `path`, `withTitle` and `withPretext`. The `mode` and `path` are just like Open Information Extraction respectively the type of the dataset (Dev, Train and Dev) and the path to the dataset. `withTitle` enables the adjustment of the title in the beginning of the text. `withPretext` gives more context to the different parts, aka with title this gives the following structure: "The claim/evidence with the title '" + title + "' says " + articleText. while if the option withTitle is not chosen, this gives "The claim/evidence says " + articleText.
Each of the functions would create a folder `text` if it isn't already made and the text editing of the claim and evidence snippets will be stored in that folder. If you prefer raw article text as input for the models, run `editArticleText` with `withTitle` and `withPretext` set to false. In the paper raw article text is used as input.  E.g. for editing the text of the claim and snippets of the Train dataset with uppercase-editing, added title and pre-text as options, run the following command:

```sh
python writeTextToDocument.py editArticleTextWithUppercaseEditing Train train.tsv true true
```
### Processing the publication date of the claim and evidence

This process is needed for predicting the verification label of claims with use of the publication date of the claim and evidence (`division1DifferencePublication`). For this the following steps are needed (should be executed in this order):

#### 1) Writing the claim date to a file
The first thing to do is writing the publication date of each claim to a file. To do this, run 
`writeClaimDate` in `writeDateToFile.py` with as arguments the type of the dataset (Dev, Train, Test) and path to that dataset. This will copy the publication time to a new file "tenses-"+mode(dev, train or test)+".txt" with a structure claimId tab publication time. E.g. to copy the publication time of each claim in the training dataset to tenses-train.txt, run the following command:

```sh
python writeDateToFile.py writeClaimDate Train train.tsv
```

#### 2) Normalising the publication time of the claim via Heideltime.
The next step is to normalise the publication date of each claim by the use of Heidetime, execute the function `normalisePublicationTimeClaim` located in `HeidelTime.java` with as argument the path to "tenses-"+mode(dev, train or test)+".txt". For each claim the timex normalisation of the publication date will be saved in a xml file with as name claimId.xml and stored in the folder `ProcessedDates`.

#### 3) Optional: Write optional publication time of snippets to file and normalise them by Heideltime.
At default, only the publication date with as structure Abbreviated month name. day, year (e.g. May 2, 2017) are considered as the publication date of the evidence. If no such date is available, it is considered that the evidence has no publication date. However an optional method is provided to search at the beginning of the evidence for a timex or for a timex near the verbs 'Published' or 'Posted'. The function `writeOptionalTimeSnippet` with as arguments the mode of the given dataset and the path to the dataset searches for those timexes and write the passages that could contain the publication date in a file with as name the snippet number in the folder processedSnippets/$claimId, e.g. processedSnippets/abbc00006/6. For doing this for each evidence of the train dataset, you could run this command:

```sh
python writeDateToFile.py writeOptionalTimeSnippet Train train.tsv
```
After the text sections are written to a file, HeidelTime will try to detect a timex and normalise those timexes in a regular form. This is done by running `normalisePublicationTimeSnippets` in HeidelTime.


### Extracting the timexes out of the text with Heideltime
This step consists out of two parts. In the first part, the publication date of the claim and evidence are gathered. These dates are used as the DCT with the news variant of Heideltime. If the publication time of the claim or evidence is not available, the narratives version of Heideltime is used. 
#### 1) Gathering the publication date of the claims or evidence
For gathering the publication date of the claims or evidence, run `writeDateToFile` in the file `writeDateToFile.py` with params the mode of the given dataset and the path to the dataset, e.g. for the Train set, this comes down to running this command:
```sh
python writeDateToFile.py writeDateToFile Train train.tsv
```
This would write the dates to data/data.txt and the claimIds and evidencenumber for linking the dates to the respective claim and evidence in data/indices.txt.
#### 2) Extracting and normalising the timexes out of the text
Next, to extracting and normalising the timexes out of the claim/evidence and writing them to a xml file in timeml format in a new folder named `processedTimes`, run `processTimexesInText` located in `HeidelTime.java` with as arguments the path to the publicationdate and indices file produced in the previous step. 

### Constructing the time bins
In this step, the construction of the time bins will be explained. The files `differencePublicationDate` and `differenceTimexesInText` describe the bins used for differentiate the claims/evidence respectevely by publication date or timexes in the text. Each line in these files introduces a bin defined by two elements: the left and right end point of the closed interval, e.g. -145 -35 defines the interval between 145 and 35 days before the publication date of the claim with -145 and -35 included. These bins are written in chronological order. The included files define the bins used in the paper, but these could also be custom made. There is also an option available to search for alernative bins where each bin contains a similar number of claim/evidence. This is suggestion is done by the use of pandas qcut. For doing this, run `BinConstructor` with either 'divisionByPublicationDate' or 'differenceTimexesInText' as first argument and then the mode and the path to the dataset as second or third parameter. This would first save the differences in respectevely the files 'differenceDaysPublicationDate.txt' and 'differenceDaysTimexesInText.txt'. Thereafter by reading that file it shows a suggestion of the bins by use of qcut. For some values of `k` bins, qcut gives an error, but you could experiment quickly with a number of `k` by uncommenting the call to `analyseExpansion1` or `analyseExpansion2` on line 639 or 643 of `BinConstructor.py`. But for doing everything together you could run for example this command for getting a suggestion for the bins when dividing by publication date when using the training dataset:
```sh
python BinConstructor.py divisionByPublicationDate Train train.tsv
```
### Training the various verification models


#### Evaluation of the models

### Running the experiments



