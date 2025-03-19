import { assets, timeData } from '@/assets/assets'
import Image from 'next/image'
import React from 'react'

const Work = ({isDarkMode}) => {
  return (
    <div id='work' className='w-full px-[12%] py-10 scroll-mt-20'>
        <h4 className='text-center mb-2 text-lg font-Ovo'>Experience</h4>
        <h2 className='text-center text-5xl font-Ovo'>Timeline</h2>

        <div className="relative mx-auto max-w-4xl px-4 py-8 space-y-5">
            {timeData.map((time, index) => (
                <div key={index} className="relative pr-8">
                <div className="relative mb-6">
                    <h3 className="text-2xl font-semibold text-gray-800 dark:text-white">
                    {time.title}
                    </h3>
                    <div className="flex justify-between items-center">
                    <h4 className="text-xl font-semibold italic text-gray-600 dark:text-gray-300">
                        {time.role}
                    </h4>
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                        {time.time}
                    </span>
                    </div>
                </div>

                {/* Description Items */}
                <div className="ml-4 space-y-6 border-l-2 border-gray-100 pl-6 dark:border-gray-700">
                    {time.description.map((desc, descIndex) => (
                    <div 
                        key={descIndex}
                        className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm transition-all hover:shadow-lg dark:border-gray-700 dark:bg-gray-800"
                    >
                        <p className="text-gray-600 dark:text-gray-300">
                        {desc}
                        </p>
                    </div>
                    ))}
                </div>
                </div>
            ))}
        </div>


    </div>
  )
}

export default Work
