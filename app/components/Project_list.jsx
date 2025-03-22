import { assets, infoList, toolsData } from '@/assets/assets'
import Image from 'next/image'
import React from 'react'

const Project_list = ({isDarkMode}) => {
    return (
        <div className="flex flex-col items-center justify-center min-h-screen ">
        <div className="w-11/12 max-w-3xl text-center mx-auto flex flex-col items-center justify-center gap-6 pt-24 pb-12">
          <h1 className="text-4xl font-bold text-gray-800 dark:text-gray-200">
            🚧 Page Under Construction 🚧
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Working hard to bring this page to life. Stay tuned!
          </p>
        </div>
        <p className="text-center max-w-2xl mx-auto mt-5 mb-12 font-serif text-gray-500 dark:text-gray-400">
          Thank you for your patience!
        </p>
      </div>
      
    )}

export default Project_list
