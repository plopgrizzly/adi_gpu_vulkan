// "adi_gpu_vulkan" crate - Licensed under the MIT LICENSE
//  * Copyright (c) 2018  Jeron A. Lau <jeron.lau@plopgrizzly.com>

// TODO: Make surface a buffer and blit onto screen with window manager.

use adi_gpu_base::WindowConnection;

use asi_vulkan;
use asi_vulkan::Vk;

pub fn create_surface(c: &mut Vk, connection: WindowConnection) {
	match connection {
		WindowConnection::Xcb(connection,window) => {
			asi_vulkan::new_surface_xcb(c, connection, window)
		}
		WindowConnection::Wayland => panic!("Wayland Rendering Not Supported Yet"),
		WindowConnection::DirectFB => panic!("DirectFB Rendering Not Supported Yet"),
		WindowConnection::Windows(connection, window) => {
			asi_vulkan::new_surface_windows(c, connection, window)
		}
		WindowConnection::Android => panic!("Android Rendering Not Supported Yet"),
		WindowConnection::IOS => panic!("IOS Rendering Not Supported Yet"),
		WindowConnection::AldaronsOS => panic!("AldaronsOS Rendering Not Supported Yet"),
		WindowConnection::Arduino => panic!("Arduino Rendering Not Supported Yet"),
		WindowConnection::Switch => panic!("Switch Rendering Not Supported Yet"),
		WindowConnection::Web => panic!("Web Assembly Rendering Not Supported Yet"),
		WindowConnection::NoOS => panic!("No OS Rendering Not Supported Yet"),
	}
}
