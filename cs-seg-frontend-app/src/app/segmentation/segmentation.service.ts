import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { SegmentedImage } from './shared/models/polygons';

@Injectable({
  providedIn: 'root',
})
export class SegmentationService {
  private readonly http = inject(HttpClient);

  uploadImage(file: File) {
    // do magic upload, get polygons json

    return this.http.get<SegmentedImage>('./polygons.json');
  }
}
