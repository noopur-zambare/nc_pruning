## Dense Network Pruning using Neural Collapse under Imbalanced Dataset

To run code:
```
python3 main.py
```


## Results
<table align="center">
  <tr>
    <td align="center">
      <h4>Testing Pruned Model</h4>
      <img src="https://github.com/noopur-zambare/nc_pruning/blob/main/testing_without_noise/10%25.png" alt="Testing Pruned Model 1" title="Testing Pruned Model" width="100%">
    </td>
    <td align="center">
      <h4>Testing Pruned Model with Noisy Data</h4>
      <img src="https://github.com/noopur-zambare/nc_pruning/blob/main/testing_with_noise/10%25.png" alt="Testing Pruned Model with Noisy Data" title="Testing Pruned Model with Noisy Data" width="100%">
    </td>
  </tr>
</table>




```
Folder Structure
├── logs
│   ├── testing_with_noise
│   ├── testing_without_noise
├── testing_with_noise
├── testing_without_noise
├── main.py
├── plots
├── README.md
├── requirements.txt
├── saves
└── utils.py
```