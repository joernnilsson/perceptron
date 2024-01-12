use image::io::Reader as ImageReader;
use na::SMatrix;

use image::imageops::resize;
use image::imageops::grayscale;

type Matrix20x20f = SMatrix<f64, 20, 20>;

pub fn matrix_to_image (matrix: &Matrix20x20f) -> image::DynamicImage {
    let mut imgbuf = image::ImageBuffer::new(20, 20);
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let val = matrix[(x as usize, y as usize)];
        let val = (val * 255.0) as u8;
        *pixel = image::Rgb([val, val, val]);
    }
    image::DynamicImage::ImageRgb8(imgbuf)
}

pub fn weights_to_image (matrix: &Matrix20x20f) -> image::DynamicImage {
    let mut imgbuf = image::ImageBuffer::new(20, 20);
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let val = matrix[(x as usize, y as usize)];
        let val = (val + 256.0/2.0) as u8;
        *pixel = image::Rgb([val, val, val]);
    }
    image::DynamicImage::ImageRgb8(imgbuf)
}

pub fn read_image(image: &String) -> Matrix20x20f {

    let img_rgb = ImageReader::open(image).unwrap().decode().unwrap().into_rgb8();
    let img_gray = grayscale(&img_rgb);
    let img_scaled = resize(&img_gray, 20, 20, image::imageops::FilterType::Nearest);

    let mut matrix = Matrix20x20f::zeros();
    for (x, y, pixel) in img_scaled.enumerate_pixels() {
        let val = pixel[0] as f64 / 255.0;
        matrix[(x as usize, y as usize)] = val;
    }
    matrix
}

pub fn read_weights(image: &String) -> Matrix20x20f {

    let img_rgb = ImageReader::open(image).unwrap().decode().unwrap().into_rgb8();
    let mut matrix = Matrix20x20f::zeros();
    for (x, y, pixel) in img_rgb.enumerate_pixels() {
        let val = pixel[0] as f64 - 256.0/2.0;
        matrix[(x as usize, y as usize)] = val;
    }
    matrix
}

pub fn save_image(matrix: &Matrix20x20f, filename: &String) {
    let img = matrix_to_image(matrix);
    img.save(filename).unwrap();
}

pub fn save_weights(matrix: &Matrix20x20f, filename: &String) {
    let img = weights_to_image(matrix);
    img.save(filename).unwrap();
}