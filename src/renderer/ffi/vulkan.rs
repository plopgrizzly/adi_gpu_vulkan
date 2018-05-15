// "adi_gpu_vulkan" crate - Licensed under the MIT LICENSE
//  * Copyright (c) 2018  Jeron A. Lau <jeron.lau@plopgrizzly.com>

use asi_vulkan;

pub struct Vulkan(pub asi_vulkan::Vk);

impl Vulkan {
	pub fn new() -> Result<Self, &'static str> {
		let connection = asi_vulkan::Vk::new();

		if let Some(c) = connection {
			Ok(Vulkan(c))
		} else {
			Err("Couldn't find Vulkan")
		}
	}
}
