
use rand::Rng;
use rayon::{prelude::*, result};

extern crate nalgebra as na;
use na::SMatrix;

// --- Types

type Matrix20x20f = SMatrix<f64, 20, 20>;

struct AnnotatedImage {
    image: Matrix20x20f,
    circle: bool, // Only 2 classes, low is rectangle, high is circle
}

struct ValidationResult {
    rectangles: u64,
    circles: u64,
    rectangles_correct: u64,
    circles_correct: u64,
}


// --- Image generators

fn make_random_rectangle() -> Matrix20x20f {
    let mut rng = rand::thread_rng();
    let xmin: u32 = rng.gen_range(0..20);
    let xmax: u32 = rng.gen_range(xmin..20);
    let ymin: u32 = rng.gen_range(0..20);
    let ymax: u32 = rng.gen_range(ymin..20);
    make_rectangle(xmin, xmax, ymin, ymax)
}

fn make_rectangle(xmin: u32, xmax: u32, ymin: u32, ymax: u32) -> Matrix20x20f {
    let mut m = Matrix20x20f::zeros();
    for x in xmin..=xmax {
        for y in ymin..=ymax {
            m[(x as usize, y as usize)] = 1.0;
        }
    }
    m
}

fn make_random_circle() -> Matrix20x20f {
    let mut rng = rand::thread_rng();

    let radius: u32 = rng.gen_range(1..10);
    let center_x: u32 = rng.gen_range(radius..(20-radius));
    let center_y: u32 = rng.gen_range(radius..(20-radius));

    make_circle(center_x, center_y, radius)
}

fn make_circle(center_x: u32, center_y: u32, radius: u32) -> Matrix20x20f {
    let mut m = Matrix20x20f::zeros();
    let radius_squared = radius.pow(2);
    for x in 0..20 {
        for y in 0..20 {
            let dist = ((x as f64 - center_x as f64).powf(2.0) + (y as f64 - center_y as f64).powf(2.0)) as f64;
            if dist <= radius_squared as f64 {
                m[(x as usize, y as usize)] = 1.0;
            }
        }
    }
    m
}

// --- Model training and validation

fn make_dataset(n: u64) -> Vec<AnnotatedImage> {
    let mut dataset: Vec<AnnotatedImage> = Vec::new();

    for _ in 0..n {
        let rect = make_random_rectangle();
        let circle = make_random_circle();

        dataset.push(AnnotatedImage{image: rect, circle: false});
        dataset.push(AnnotatedImage{image: circle, circle: true});
    }

    dataset
}

fn train_dataset(dataset: &Vec<AnnotatedImage>, n: u64) -> Matrix20x20f {
    let mut weights = Matrix20x20f::zeros();

    for i in 0..n{
        let mut weights_changed: bool = false;
        for image in dataset {
            let output = image.image.dot(&weights);
            if image.circle && output < 0.5 {
                weights = weights + image.image;
                weights_changed = true;
            } else if !image.circle && output > 0.5 {
                weights = weights - image.image;
                weights_changed = true;
            }
        }
        if !weights_changed {
            println!("Converged after {} iterations", i);
            break;
        }

        // Print progress
        let validation_result = validate_dataset(&dataset, weights);
        println!("Iteration: {} rectangles: {}/{} circles: {}/{}", i, validation_result.rectangles_correct, validation_result.rectangles, validation_result.circles_correct, validation_result.circles);
    }

    weights
}

fn validate_dataset(dataset: &Vec<AnnotatedImage>, weights: Matrix20x20f) -> ValidationResult {
    let mut rectangle_count: u64 = 0;
    let mut circle_count: u64 = 0;
    let mut rectangle_correct: u64 = 0;
    let mut circle_correct: u64 = 0;

    for image in dataset {
        let output = image.image.dot(&weights);

        if image.circle {
            circle_count += 1;
        } else {
            rectangle_count += 1;
        }

        if image.circle && output > 0.5 {
            circle_correct += 1;
        } else if !image.circle && output < 0.5 {
            rectangle_correct += 1;
        }
    }

    ValidationResult {
        rectangles: rectangle_count,
        circles: circle_count,
        rectangles_correct: rectangle_correct,
        circles_correct: circle_correct,
    }
}

fn main(){

    // Make random dataset of circles and rectangles
    let dataset = make_dataset(2000);

    // Train model using dataset
    println!("--------------- Training -----------------");
    let weights = train_dataset(&dataset, 1000);

    // Verify model using dataset
    validate_dataset(&dataset, weights);

    // Validate using new dataset of random circles and rectangles
    let validation_dataset = make_dataset(20);
    let validation_result = validate_dataset(&validation_dataset, weights);
    
    println!("---------------- Validation ---------------");
    println!("Rectangles correct: {}/{}", validation_result.rectangles_correct, validation_result.rectangles);
    println!("Circles correct: {}/{}", validation_result.circles_correct, validation_result.circles);


}
