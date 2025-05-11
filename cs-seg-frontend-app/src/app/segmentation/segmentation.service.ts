import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root',
})
export class SegmentationService {
  private readonly http = inject(HttpClient);

  baseUrl = 'http://localhost:5000/';

  uploadImage(file: File, skip_labels: string[]) {
    const skip_labels_string = skip_labels
      .map((label) => `skip_labels=${label}`)
      .join('&');

    let url = `${this.baseUrl}/images`;

    if (skip_labels_string) {
      url += `?${skip_labels_string}`;
    }

    const formData = new FormData();
    formData.append('image', file);

    return this.http.post<Blob>(url, formData, {
      responseType: 'blob' as 'json',
    });
  }
}
