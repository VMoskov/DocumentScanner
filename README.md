# DocumentScanner
A simple document scanner built using OpenCV and imutils<br>


## Installation
1. Clone the repository
```bash
git clone https://github.com/VMoskov/DocumentScanner.git
```
2. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage
1. Run the script with the following command
```bash
python document_scanner.py --image images/<file_name>.jpg
```
or just
```bash
python document_scanner.py -i images/<file_name>.jpg
```
2. The scanned document will be displayed on the screen
3. Press `0` to close the window

## Example
Original image:<br>
![Original Image](results/receipt_image.png)<br>
Image with edges:<br>
![Image with Edges](results/receipt_edges.png)<br>
Detected document:<br>
![Detected Document](results/receipt_paper_contour.png)<br>
Scanned document:<br>
![Scanned Document](results/receipt_scan.png)

## More examples with real-life documents
1. Paper with text:<br>
   ![Original Image](results/document_image.png)<br>
   ![Image with Edges](results/document_edges.png)<br>
    ![Detected Document](results/document_paper_contour.png)<br>
    ![Scanned Document](results/document_scan.png)

2. Colored payment slip:<br>
    ![Original Image](results/payment_slip_image.png)<br>
    ![Image with Edges](results/payment_slip_edges.png)<br>
     ![Detected Document](results/payment_slip_paper_contour.png)<br>
     ![Scanned Document](results/payment_slip_scan.png)