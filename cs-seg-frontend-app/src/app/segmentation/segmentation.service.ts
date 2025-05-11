import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { SegmentedImage } from './shared/models/polygons';

@Injectable({
  providedIn: 'root',
})
export class SegmentationService {
  private readonly http = inject(HttpClient);

  baseUrl = 'http://localhost:5000/';

  uploadImage(file: File, skip_labels: string[]) {
    // do magic upload, get polygons json

    return this.http.get<SegmentedImage>(`${this.baseUrl}/images?`);
  }
}
