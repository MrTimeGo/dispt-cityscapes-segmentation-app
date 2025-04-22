export interface SegmentedImage {
  imgHeight: number;
  imgWidth: number;
  objects: Polygon[];
}

export interface Polygon {
  label: string;
  polygon: number[][];
}
