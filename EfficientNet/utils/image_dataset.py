class ImageScoreDataset(Dataset):
    def __init__(self, labels_df=None, csv_file=None, image_folder=None, transform=None):
        if csv_file:
            self.labels = pd.read_csv(csv_file)
        else:
            self.labels = labels_df
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_id = str(self.labels.iloc[idx]['id'])
        score = float(self.labels.iloc[idx]['score'])
        
        # Load image
        image_path = f"{self.image_folder}/{image_id}.tif"
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms (if any)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(score)
