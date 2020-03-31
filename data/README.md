# CORD-19 processed data

### Download

Tsinghua Cloud
```
https://cloud.tsinghua.edu.cn/f/ac0352b7c0054f5f9c7d/
```
Google Driver
```
https://drive.google.com/open?id=1L2rM5tZAcHFgmuBnBWwi27XVRGkH9eK0
```
Collections
```
https://cloud.tsinghua.edu.cn/d/a801f337a3b14892a138/
```
### Data Satistics

There are four files as follow

|Index|Filename|Merged From|How many Papers|
|:---:|:------:|:----------:|:--:|
|1|comm_use_subset.jsonl|Commercial use subset (includes PMC content)|9000|
|2|noncomm_use_subset.jsonl|Non-commercial use subset (includes PMC content)|1973|
|3|pmc_custom_license_subset.jsonl|PMC custom license subset|1426|
|4|biorxiv_medrxiv_subset.jsonl|bioRxiv/medRxiv subset (pre-prints that are not peer reviewed)|803|

### Data Format

- One paper per line.
- Each paper is a dict and has keys:
  - title
  - paper_id
  - abstract : a list of paragraphs 
  - body_text: body text 
  - back_matter: additional information

