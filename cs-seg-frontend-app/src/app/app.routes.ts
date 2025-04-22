import { Routes } from '@angular/router';
import { SegmentationComponent } from './segmentation/segmentation.component';

export const routes: Routes = [
  { path: 'segmentation', component: SegmentationComponent },
  { path: '', redirectTo: 'segmentation', pathMatch: 'full' },
];
