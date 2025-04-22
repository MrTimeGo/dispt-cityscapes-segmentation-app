import { Component, inject, signal } from '@angular/core';
import { ToolbarModule } from 'primeng/toolbar';
import { FileSelectEvent, FileUploadModule } from 'primeng/fileupload';
import { ButtonModule } from 'primeng/button';
import { SegmentationService } from './segmentation.service';
import { SegmentedImage } from './shared/models/polygons';
import { ImageCompareModule } from 'primeng/imagecompare';

@Component({
  selector: 'app-segmentation',
  imports: [ToolbarModule, FileUploadModule, ButtonModule, ImageCompareModule],
  templateUrl: './segmentation.component.html',
  styleUrl: './segmentation.component.css',
})
export class SegmentationComponent {
  private readonly segmentationService = inject(SegmentationService);

  image: File | null = null;
  imageUrl1: string | null = null;
  imageUrl2: string | null = null;

  colors = {
    road: '#808080', // gray
    sidewalk: '#C0C0C0', // light gray
    sky: '#87CEEB', // sky blue
    car: '#1E90FF', // dodger blue
    terrain: '#8B4513', // saddle brown
    building: '#CD853F', // peru (brownish)
    vegetation: '#228B22', // forest green
    pole: '#4A4A4A', // dark gray
    'traffic sign': '#FFD700', // gold
    static: '#000000', // black
    bicycle: '#B8860B', // dark goldenrod
    person: '#FF69B4', // hot pink
    'license plate': '#FFFFFF', // white
    rider: '#9370DB', // medium purple
    'ego vehicle': '#4682B4', // steel blue
    'out of roi': '#2F4F4F', // dark slate gray
  };

  onFileSelected(event: FileSelectEvent) {
    console.log('onFileSelected', event);
    this.image = event.currentFiles[0];
    // Create a URL for the selected image file
    if (this.image) {
      this.imageUrl1 = URL.createObjectURL(this.image);
      this.segmentationService.uploadImage(this.image).subscribe((polygons) => {
        this.drawPolygons(polygons);
      });
    }
  }

  private drawPolygons(image: SegmentedImage) {
    const canvas = document.createElement('canvas');
    canvas.width = image.imgWidth;
    canvas.height = image.imgHeight;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      return;
    }

    for (let i = 0; i < image.objects.length; i++) {
      const object = image.objects[i];
      const color = this.colors[object.label as keyof typeof this.colors];

      ctx.strokeStyle = color;
      ctx.fillStyle = color;

      ctx.beginPath();

      for (const point of object.polygon) {
        ctx.lineTo(point[0], point[1]);
      }

      ctx.lineTo(object.polygon[0][0], object.polygon[0][1]);

      ctx.stroke();
      ctx.fill();
    }

    this.imageUrl2 = canvas.toDataURL();
    canvas.remove();
  }

  // Clean up the object URL when component is destroyed
  ngOnDestroy() {
    if (this.imageUrl1) {
      URL.revokeObjectURL(this.imageUrl1);
    }
  }
}
