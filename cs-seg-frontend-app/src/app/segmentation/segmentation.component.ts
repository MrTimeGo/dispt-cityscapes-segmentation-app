import { Component, inject, signal } from '@angular/core';
import { ToolbarModule } from 'primeng/toolbar';
import { FileSelectEvent, FileUploadModule } from 'primeng/fileupload';
import { ButtonModule } from 'primeng/button';
import { SegmentationService } from './segmentation.service';
import { ImageCompareModule } from 'primeng/imagecompare';
import { MultiSelectChangeEvent, MultiSelectModule } from 'primeng/multiselect';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-segmentation',
  imports: [
    ToolbarModule,
    FileUploadModule,
    ButtonModule,
    ImageCompareModule,
    MultiSelectModule,
    FormsModule,
  ],
  templateUrl: './segmentation.component.html',
  styleUrl: './segmentation.component.css',
})
export class SegmentationComponent {
  private readonly segmentationService = inject(SegmentationService);

  image: File | null = null;
  imageUrl1: string | null = null;
  imageUrl2: string | null = null;

  allLabels: string[] = [
    'road',
    'sidewalk',
    'person',
    'car',
    'truck',
    'bus',
    'motorcycle',
    'bicycle',
    'building',
    'vegetation',
    'terrain',
    'sky',
    'parking',
  ];

  selectedLabels = [...this.allLabels];

  onFileSelected(event: FileSelectEvent) {
    console.log('onFileSelected', event);
    this.image = event.currentFiles[0];
    // Create a URL for the selected image file
    if (this.image) {
      this.imageUrl1 = URL.createObjectURL(this.image);

      // difference between selectedLabels and allLabels
      const skipLabels = this.allLabels.filter(
        (label) => !this.selectedLabels.includes(label),
      );

      this.segmentationService
        .uploadImage(this.image, skipLabels)
        .subscribe((img) => {
          this.imageUrl2 = URL.createObjectURL(img);
        });
    }
  }

  onChange() {
    const skipLabels = this.allLabels.filter(
      (label) => !this.selectedLabels.includes(label),
    );

    this.segmentationService
      .uploadImage(this.image!, skipLabels)
      .subscribe((img) => {
        this.imageUrl2 = URL.createObjectURL(img);
      });
  }

  onExportSegmentation() {
    // download the imageUrl2
    const a = document.createElement('a');
    a.href = this.imageUrl2!;
    a.download = `${this.image!.name}-segmentation.png`;
    a.click();
  }

  // Clean up the object URL when component is destroyed
  ngOnDestroy() {
    if (this.imageUrl1) {
      URL.revokeObjectURL(this.imageUrl1);
    }
  }
}
