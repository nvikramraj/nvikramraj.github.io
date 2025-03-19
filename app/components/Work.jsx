import { assets, serviceData } from '@/assets/assets'
import Image from 'next/image'
import React from 'react'

const Work = ({isDarkMode}) => {
  return (
    <div id='work' className='w-full px-[12%] py-10 scroll-mt-20'>
        <h4 className='text-center mb-2 text-lg font-Ovo'>Work Experience</h4>
        <h2 className='text-center text-5xl font-Ovo'>Timeline</h2>
        <p className='text-center max-w-2xl mx-auto mt-5 mb-12 font-Ovo'>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec ac nulla sodales, dapibus dui ac, 
            aliquam ipsum. Cras blandit arcu sed luctus mattis. Praesent vel fringilla neque. Donec non sodales 
            lectus, non rutrum sem. Nulla fringilla quam lectus, eget elementum tellus pharetra id. Praesent 
            elementum sed est non placerat.
        </p>

        <div className='grid grid-cols-auto gap-6 my-10'>
                {serviceData.map(({icon,title,description,link},index)=>(
                    <div key={index} 
                    className='border border-gray-400 rounded-lg px-8 py-12 hover:shadow-black cursor-pointer hover:bg-lightHover
                    hover:-translate-y-1 duration-500 dark:hover:bg-darkHover dark:hover:shadow-white'>
                        <Image src={icon} alt='' className='w-10' />
                        <h3 className='test-lg my-4 text-gray-700 dark:text-white'>{title}</h3>
                        <p className='text-sm text-gray-600 leading-5 dark:text-white/80'>
                            {description}
                        </p>
                        <a href={link} className='flex items-center gap-2 text-sm mt-5'>
                            Read more <Image src={assets.right_arrow} alt='' className='w-4'/>
                        </a>
                    </div>
                ))}
            </div>


        <a href="" className='w-max flex items-center justify-center gap-2 text-gray-700 border-[0.5px] border-gray-700 rounded-full py-3 px-10 mx-auto
        my-20 hover:bg-lightHover duration-500 dark:text-white dark:border-white dark:hover:bg-darkHover'>
            Show more <Image src={isDarkMode ? assets.right_arrow_bold_dark : assets.right_arrow_bold} alt='Right arrow' className='w-4' />
        </a>

    </div>
  )
}

export default Work
