// Aldaron's Device Interface / GPU / Vulkan
// Copyright (c) 2017-2018 Jeron Lau <jeron.lau@plopgrizzly.com>
// Licensed under the MIT LICENSE
//
// src/lib.rs

//! Aldaron's Device Interface / GPU is a library developed by Plop Grizzly for
//! interfacing with the GPU to render graphics or do fast calculations.

// #![no_std]

#[macro_use]
extern crate ami;
extern crate awi;
extern crate afi;
extern crate asi_vulkan;
extern crate adi_gpu_base;
extern crate libc;

/// Transform represents a transformation matrix.
pub(crate) mod renderer;

use ami::*;
use adi_gpu_base as base;

pub use renderer::Texture;

/// To render anything with adi_gpu, you have to make a `Display`
pub struct Display {
	renderer: renderer::Renderer,
}

impl base::Display for Display {
	type Model = Model;
	type Texture = Texture;
	type Gradient = Gradient;
	type TexCoords = TexCoords;
	type Shape = Shape;

	fn new(window: &awi::Window) -> Option<Self> {
		let renderer = renderer::Renderer::new("ADI Application",
			window.get_connection(), (0.0, 0.0, 0.0));

		Some(Display { renderer })
	}

	fn color(&mut self, color: (f32, f32, f32)) {
		self.renderer.bg_color(color);
	}

	fn update(&mut self) {
		self.renderer.update();
	}

	fn camera(&mut self, xyz: (f32,f32,f32), rotate_xyz: (f32,f32,f32)) {
		self.renderer.set_camera(xyz, rotate_xyz);
		self.renderer.camera();
	}

	fn model(&mut self, vertices: &[f32], indices: &[u32]) -> Model {
		Model(self.renderer.model(vertices, indices))
	}

	fn fog(&mut self, fog: Option<(f32, f32)>) -> () {
		if let Some(fog) = fog {
			self.renderer.fog(fog);
		} else {
			self.renderer.fog((::std::f32::MAX, 0.0));
		}
	}

	fn texture(&mut self, graphic: afi::Graphic) -> Texture {
		let (w, h, pixels) = graphic.as_slice();

		self.renderer.texture(w, h, pixels)
	}

	fn gradient(&mut self, colors: &[f32]) -> Gradient {
		Gradient(self.renderer.colors(colors))
	}

	fn texcoords(&mut self, texcoords: &[f32]) -> TexCoords {
		TexCoords(self.renderer.texcoords(texcoords))
	}

	fn set_texture(&mut self, texture: &mut Self::Texture, pixels: &[u32]) {
		self.renderer.set_texture(texture, pixels);
	}

	#[inline(always)]
	fn shape_solid(&mut self, model: &Model, transform: Mat4,
		color: [f32; 4], blending: bool, fancy: bool, fog: bool,
		camera: bool) -> Shape
	{
		Shape(self.renderer.solid(model.0, transform.0, color,
			blending, fancy, fog, camera))
	}

	#[inline(always)]
	fn shape_gradient(&mut self, model: &Model, transform: Mat4,
		colors: Gradient, blending: bool, fancy: bool, fog: bool,
		camera: bool) -> Shape
	{
		Shape(self.renderer.gradient(model.0, transform.0,
			colors.0, blending, fancy, fog, camera))
	}

	#[inline(always)]
	fn shape_texture(&mut self, model: &Model, transform: Mat4,
		texture: Texture, tc: TexCoords, blending: bool, fancy: bool,
		fog: bool, camera: bool) -> Shape
	{
		Shape(self.renderer.textured(model.0, transform.0,
			texture, tc.0, blending, fancy, fog, camera))
	}

	#[inline(always)]
	fn shape_faded(&mut self, model: &Model, transform: Mat4,
		texture: Texture, tc: TexCoords, alpha: f32, fancy: bool,
		fog: bool, camera: bool) -> Shape
	{
		Shape(self.renderer.faded(model.0, transform.0,
			texture, tc.0, alpha, fancy, fog, camera))
	}

	#[inline(always)]
	fn shape_tinted(&mut self, model: &Model, transform: Mat4,
		texture: Texture, tc: TexCoords, tint: [f32; 4], blending: bool,
		fancy: bool, fog: bool, camera: bool) -> Shape
	{
		Shape(self.renderer.tinted(model.0, transform.0,
			texture, tc.0, tint, blending, fancy, fog, camera))
	}

	#[inline(always)]
	fn shape_complex(&mut self, model: &Model, transform: Mat4,
		texture: Texture, tc: TexCoords, tints: Gradient,
		blending: bool, fancy: bool, fog: bool, camera: bool) -> Shape
	{
		Shape(self.renderer.complex(model.0, transform.0,
			texture, tc.0, tints.0, blending, fancy, fog, camera))
	}

	fn transform(&mut self, shape: &mut Self::Shape, transform: &Mat4) {
		self.renderer.transform(&mut shape.0, transform);
	}

	fn resize(&mut self, wh: (u32, u32)) -> () {
		self.renderer.resize(wh);
	}
}

/// A list of vertices that make a shape.
#[derive(Copy, Clone)]
pub struct Model(usize);

impl base::Model for Model {
}

/// A list of colors to be paired with vertices.
#[derive(Copy, Clone)]
pub struct Gradient(usize);

impl base::Gradient for Gradient {
}

/// A list of texture coordinates to be paired with vertices.
#[derive(Copy, Clone)]
pub struct TexCoords(usize);

impl base::TexCoords for TexCoords {
}

impl base::Texture for Texture {
	/// Get the width and height.
	fn wh(&self) -> (u32, u32) {
		(self.w, self.h)
	}
}

/// A renderable object that exists on the `Display`.
pub struct Shape(renderer::ShapeHandle);

impl base::Shape for Shape {
}
